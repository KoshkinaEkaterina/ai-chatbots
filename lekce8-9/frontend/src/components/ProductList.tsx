import React from 'react';
import { Product } from '../types';

interface ProductListProps {
    products: Product[];
    criteria?: Record<string, any>;
}

export default function ProductList({ products, criteria }: ProductListProps) {
    return (
        <div className="space-y-4">
            {criteria && (
                <div className="bg-blue-50 p-4 rounded-lg">
                    <h3 className="font-bold mb-2">Search Criteria:</h3>
                    <div className="space-y-1">
                        {criteria.price_range && (
                            <p>Price: 
                                {criteria.price_range.min && `$${criteria.price_range.min}`}
                                {criteria.price_range.min && criteria.price_range.max && ' - '}
                                {criteria.price_range.max && `$${criteria.price_range.max}`}
                            </p>
                        )}
                        {criteria.category && <p>Category: {criteria.category}</p>}
                        {criteria.features?.length > 0 && (
                            <p>Features: {criteria.features.join(', ')}</p>
                        )}
                    </div>
                </div>
            )}
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {products.map((product) => (
                    <div key={product.id} className="border rounded-lg p-4 hover:shadow-lg transition-shadow">
                        <h3 className="font-bold text-lg">{product.name}</h3>
                        <p className="text-xl text-blue-600 font-bold">${product.price.toFixed(2)}</p>
                        <p className="text-sm text-gray-600">
                            {product.category.join(' > ')}
                        </p>
                        <p className="mt-2">{product.description}</p>
                        {Object.keys(product.dimensions).length > 0 && (
                            <div className="mt-2 text-sm">
                                <p className="font-semibold">Dimensions:</p>
                                {Object.entries(product.dimensions).map(([key, value]) => (
                                    <span key={key} className="mr-4">
                                        {key}: {value}cm
                                    </span>
                                ))}
                            </div>
                        )}
                        {product.weight && (
                            <p className="text-sm mt-1">Weight: {product.weight}kg</p>
                        )}
                        <div className="mt-2 text-sm text-gray-500">
                            Match score: {(product.confidence * 100).toFixed(1)}%
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
} 